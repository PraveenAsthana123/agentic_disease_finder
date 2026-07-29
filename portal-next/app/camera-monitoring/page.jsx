'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const qualityColor = q =>
  q === 'excellent' ? 'success' :
  q === 'good'      ? 'primary' :
  q === 'fair'      ? 'warning' : 'danger';

const statusColor = s =>
  s === 'completed'   ? 'success' :
  s === 'active'      ? 'primary' :
  s === 'interrupted' ? 'warning' : 'danger';

const typeColor = t =>
  t === 'continuous'     ? 'primary' :
  t === 'event_triggered'? 'warning' : 'secondary';

export default function CameraMonitoringPage() {
  const [ov,   setOv]   = useState(null);
  const [bd,   setBd]   = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab,  setTab]  = useState('overview');

  useEffect(() => {
    fetch(`${API}/api/camera-monitoring/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/camera-monitoring/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/camera-monitoring/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const kpis = ov.kpis || {};
  const tabs = [
    { id: 'overview',    label: 'Overview' },
    { id: 'sessions',    label: 'Sessions' },
    { id: 'by-location', label: 'By Location' },
    { id: 'definitions', label: 'Definitions' },
  ];

  return (
    <div>
      <h3>&#x1f4f9; Camera Monitoring Dashboard</h3>
      <p className="text-muted small">
        Video-based seizure monitoring across 78 sessions and 27 patients. Covers home and EMU
        (Epilepsy Monitoring Unit) cameras — seizure detection, false alarm rates, recording
        quality, night vision usage, and response times.
      </p>

      {/* KPI cards */}
      <div className="row mb-3">
        {[
          { label: 'Total Sessions',       value: kpis.total_sessions,                           color: 'primary' },
          { label: 'Patients Monitored',   value: kpis.total_patients,                            color: 'info' },
          { label: 'Seizure Events',        value: kpis.total_seizure_events,                      color: 'danger' },
          { label: 'Movement Events',       value: kpis.total_movement_events,                     color: 'warning' },
          { label: 'False Alarms',          value: kpis.total_false_alarms,                        color: 'secondary' },
          { label: 'Avg Duration (hrs)',    value: kpis.avg_duration_hours,                        color: 'primary' },
          { label: 'Alert Rate %',          value: `${kpis.alert_rate}%`,                          color: kpis.alert_rate > 60 ? 'warning' : 'success' },
          { label: 'Seizure Detection %',   value: `${kpis.seizure_detection_rate}%`,              color: 'danger' },
          { label: 'Night Vision Rate %',   value: `${kpis.night_vision_rate}%`,                   color: 'info' },
          { label: 'Avg Response (sec)',    value: `${Math.round(kpis.avg_response_time_seconds)}s`, color: kpis.avg_response_time_seconds > 300 ? 'warning' : 'success' },
        ].map(c => (
          <div key={c.label} className="col-6 col-md-3 col-lg-2 mb-2">
            <div className="card text-center shadow-sm border-0">
              <div className="card-body py-2 px-1">
                <div className={`h4 mb-0 text-${c.color}`}>{c.value ?? '—'}</div>
                <div className="text-muted" style={{fontSize: '0.70rem'}}>{c.label}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {tabs.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* ── Overview Tab ─────────────────────────────────────────── */}
      {tab === 'overview' && (
        <div className="row">
          {/* Location Distribution */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold">Camera Location</div>
              <div className="card-body">
                {(ov.location_dist || []).map((loc, i) => {
                  const pct = kpis.total_sessions ? Math.round(loc.count / kpis.total_sessions * 100) : 0;
                  const label = loc.location === 'emu_room' ? 'EMU Room' :
                                loc.location.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
                  return (
                    <div key={i} className="d-flex align-items-center mb-2">
                      <span className="small" style={{minWidth: 90}}>{label}</span>
                      <div className="flex-grow-1 mx-2">
                        <div className="progress" style={{height: 20}}>
                          <div className="progress-bar bg-primary" style={{width: `${Math.max(6, pct)}%`}}>
                            {loc.count}
                          </div>
                        </div>
                      </div>
                      <span className="small text-muted">{pct}%</span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Recording Quality */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold">Recording Quality</div>
              <div className="card-body">
                {(ov.quality_dist || []).map((q, i) => {
                  const pct = kpis.total_sessions ? Math.round(q.count / kpis.total_sessions * 100) : 0;
                  return (
                    <div key={i} className="d-flex align-items-center mb-2">
                      <span className={`badge bg-${qualityColor(q.quality)} me-2`} style={{minWidth: 72}}>
                        {q.quality}
                      </span>
                      <div className="flex-grow-1 mx-2">
                        <div className="progress" style={{height: 20}}>
                          <div className={`progress-bar bg-${qualityColor(q.quality)}`}
                               style={{width: `${Math.max(6, pct)}%`}}>
                            {q.count}
                          </div>
                        </div>
                      </div>
                      <span className="small text-muted">{pct}%</span>
                    </div>
                  );
                })}
                <div className="alert alert-warning small mt-2 mb-0 py-2">
                  <strong>{(ov.quality_dist || []).filter(q => q.quality === 'poor').reduce((s, q) => s + q.count, 0)}</strong> sessions with poor quality — may affect detection accuracy
                </div>
              </div>
            </div>
          </div>

          {/* Session Type & Status */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm h-100">
              <div className="card-header fw-bold">Session Type & Status</div>
              <div className="card-body">
                <div className="small fw-semibold text-muted mb-2">Session Type</div>
                {(ov.session_type_dist || []).map((t, i) => {
                  const pct = kpis.total_sessions ? Math.round(t.count / kpis.total_sessions * 100) : 0;
                  const label = t.type.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
                  return (
                    <div key={i} className="d-flex align-items-center mb-2">
                      <span className={`badge bg-${typeColor(t.type)} me-2`} style={{minWidth: 108, fontSize: '0.68rem'}}>
                        {label}
                      </span>
                      <div className="flex-grow-1 mx-1">
                        <div className="progress" style={{height: 16}}>
                          <div className={`progress-bar bg-${typeColor(t.type)}`}
                               style={{width: `${Math.max(6, pct)}%`}}>
                            {t.count}
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
                <hr className="my-2" />
                <div className="small fw-semibold text-muted mb-2">Status</div>
                {(ov.status_dist || []).map((s, i) => {
                  const pct = kpis.total_sessions ? Math.round(s.count / kpis.total_sessions * 100) : 0;
                  return (
                    <div key={i} className="d-flex align-items-center mb-1">
                      <span className={`badge bg-${statusColor(s.status)} me-2`} style={{minWidth: 80}}>
                        {s.status}
                      </span>
                      <span className="small text-muted">{s.count} ({pct}%)</span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Monthly Trend */}
          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Monthly Trend — Sessions &amp; Events</div>
              <div className="card-body">
                <div className="table-responsive">
                  <table className="table table-sm table-bordered mb-0">
                    <thead className="table-dark">
                      <tr>
                        <th>Month</th>
                        <th>Sessions</th>
                        <th>Seizure Events</th>
                        <th>Movement Events</th>
                        <th>False Alarms</th>
                        <th>Seizure Rate</th>
                        <th>False Alarm Rate</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(ov.monthly_trend || []).map((m, i) => {
                        const totalEvents = m.seizure_events + m.movement_events + m.false_alarms;
                        const seizureRate = totalEvents > 0 ? Math.round(m.seizure_events / totalEvents * 100) : 0;
                        const faRate = m.sessions > 0 ? Math.round(m.false_alarms / m.sessions * 100) : 0;
                        return (
                          <tr key={i}>
                            <td className="fw-semibold">{m.month}</td>
                            <td>{m.sessions}</td>
                            <td className="text-danger fw-bold">{m.seizure_events}</td>
                            <td className="text-warning">{m.movement_events}</td>
                            <td className="text-secondary">{m.false_alarms}</td>
                            <td>
                              <div className="progress" style={{height: 16, minWidth: 80}}>
                                <div className={`progress-bar ${seizureRate > 40 ? 'bg-danger' : 'bg-success'}`}
                                     style={{width: `${seizureRate}%`}}>
                                  {seizureRate}%
                                </div>
                              </div>
                            </td>
                            <td>
                              <div className="progress" style={{height: 16, minWidth: 80}}>
                                <div className={`progress-bar ${faRate > 50 ? 'bg-warning' : 'bg-success'}`}
                                     style={{width: `${Math.max(4, faRate)}%`}}>
                                  {faRate}%
                                </div>
                              </div>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Sessions Tab ──────────────────────────────────────────── */}
      {tab === 'sessions' && bd && (
        <div>
          <div className="card shadow-sm">
            <div className="card-header fw-bold">
              All Sessions ({(bd.sessions || []).length} records — sorted by date desc)
            </div>
            <div className="card-body p-0">
              <div style={{maxHeight: 560, overflowY: 'auto'}}>
                <table className="table table-sm table-hover mb-0">
                  <thead className="table-dark sticky-top">
                    <tr>
                      <th>Patient</th>
                      <th>Date</th>
                      <th>Location</th>
                      <th>Type</th>
                      <th>Duration</th>
                      <th>Seizures</th>
                      <th>Moves</th>
                      <th>F.Alarms</th>
                      <th>Alert</th>
                      <th>Response</th>
                      <th>Quality</th>
                      <th>Night&#x1f319;</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(bd.sessions || []).map((s, i) => {
                      const locLabel = s.camera_location === 'emu_room' ? 'EMU' :
                                       s.camera_location.replace('_', ' ');
                      return (
                        <tr key={i}>
                          <td className="fw-semibold small">{s.patient_id}</td>
                          <td className="small">{s.session_date}</td>
                          <td className="small">{locLabel}</td>
                          <td><span className={`badge bg-${typeColor(s.session_type)}`} style={{fontSize:'0.65rem'}}>
                            {s.session_type.replace('_',' ')}
                          </span></td>
                          <td className="small">{s.duration_hours}h</td>
                          <td className={s.seizure_events > 0 ? 'text-danger fw-bold' : ''}>{s.seizure_events}</td>
                          <td>{s.movement_events}</td>
                          <td className={s.false_alarms > 0 ? 'text-warning' : ''}>{s.false_alarms}</td>
                          <td>{s.alert_sent ? '&#x2705;' : '&#x2796;'}</td>
                          <td className="small">{s.response_time_seconds ? `${s.response_time_seconds}s` : '—'}</td>
                          <td><span className={`badge bg-${qualityColor(s.recording_quality)}`} style={{fontSize:'0.62rem'}}>
                            {s.recording_quality}
                          </span></td>
                          <td>{s.night_vision ? '&#x1f319;' : '—'}</td>
                          <td><span className={`badge bg-${statusColor(s.status)}`} style={{fontSize:'0.62rem'}}>
                            {s.status}
                          </span></td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Per-patient summary */}
          {(bd.by_patient || []).length > 0 && (
            <div className="card shadow-sm mt-3">
              <div className="card-header fw-bold">Per-Patient Summary ({(bd.by_patient || []).length} patients)</div>
              <div className="card-body p-0">
                <div style={{maxHeight: 360, overflowY: 'auto'}}>
                  <table className="table table-sm table-striped mb-0">
                    <thead className="table-dark sticky-top">
                      <tr>
                        <th>Patient</th>
                        <th>Sessions</th>
                        <th>Total Hours</th>
                        <th>Seizure Events</th>
                        <th>Movement Events</th>
                        <th>False Alarms</th>
                        <th>Avg Response (s)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(bd.by_patient || []).map((p, i) => (
                        <tr key={i}>
                          <td className="fw-semibold small">{p.patient_id}</td>
                          <td>{p.sessions}</td>
                          <td>{p.total_duration}h</td>
                          <td className={p.seizure_events > 0 ? 'text-danger fw-bold' : ''}>{p.seizure_events}</td>
                          <td>{p.movement_events}</td>
                          <td className={p.false_alarms > 0 ? 'text-warning' : ''}>{p.false_alarms}</td>
                          <td>{p.avg_response_time ? `${Math.round(p.avg_response_time)}s` : '—'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── By Location Tab ───────────────────────────────────────── */}
      {tab === 'by-location' && bd && (
        <div className="row">
          {(bd.by_location || []).map((loc, i) => {
            const label = loc.location === 'emu_room' ? 'EMU Room (Epilepsy Monitoring Unit)' :
                          loc.location.replace('_', ' ').replace(/\b\w/g, c => c.toUpperCase());
            const totalEvents = loc.seizure_events + loc.movement_events + loc.false_alarms;
            const seizurePct = totalEvents > 0 ? Math.round(loc.seizure_events / totalEvents * 100) : 0;
            const faPct = loc.sessions > 0 ? Math.round(loc.false_alarms / loc.sessions * 100) : 0;
            const qbTotal = Object.values(loc.quality_breakdown || {}).reduce((s, v) => s + v, 0);
            return (
              <div key={i} className="col-md-6 mb-3">
                <div className="card shadow-sm">
                  <div className="card-header fw-bold">
                    &#x1f4f9; {label} <span className="badge bg-secondary ms-2">{loc.sessions} sessions</span>
                  </div>
                  <div className="card-body">
                    <div className="row mb-2">
                      {[
                        { label: 'Avg Duration', value: `${loc.avg_duration}h`, color: 'primary' },
                        { label: 'Seizure Events', value: loc.seizure_events, color: 'danger' },
                        { label: 'Movement Events', value: loc.movement_events, color: 'warning' },
                        { label: 'False Alarms', value: loc.false_alarms, color: 'secondary' },
                      ].map(c => (
                        <div key={c.label} className="col-6 mb-2">
                          <div className={`text-${c.color} fw-bold`}>{c.value}</div>
                          <div className="text-muted" style={{fontSize: '0.70rem'}}>{c.label}</div>
                        </div>
                      ))}
                    </div>
                    <div className="mb-2">
                      <div className="small text-muted mb-1">Seizure detection rate (of events)</div>
                      <div className="progress" style={{height: 18}}>
                        <div className={`progress-bar ${seizurePct > 50 ? 'bg-danger' : 'bg-success'}`}
                             style={{width: `${Math.max(4, seizurePct)}%`}}>
                          {seizurePct}%
                        </div>
                      </div>
                    </div>
                    <div className="mb-2">
                      <div className="small text-muted mb-1">False alarm rate (per session)</div>
                      <div className="progress" style={{height: 18}}>
                        <div className={`progress-bar ${faPct > 40 ? 'bg-warning' : 'bg-success'}`}
                             style={{width: `${Math.max(4, faPct)}%`}}>
                          {faPct}%
                        </div>
                      </div>
                    </div>
                    <div>
                      <div className="small text-muted mb-1">Recording Quality Breakdown</div>
                      <div className="d-flex gap-2 flex-wrap">
                        {Object.entries(loc.quality_breakdown || {}).map(([q, cnt]) => (
                          <span key={q} className={`badge bg-${qualityColor(q)}`}>
                            {q}: {cnt}
                          </span>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* ── Definitions Tab ───────────────────────────────────────── */}
      {tab === 'definitions' && defs && (
        <div>
          <div className="card shadow-sm mb-3 border-primary">
            <div className="card-header fw-bold bg-primary text-white">Dashboard Purpose</div>
            <div className="card-body small">
              Camera Monitoring Sessions Dashboard — tracks video-based seizure monitoring
              for 27 epilepsy patients across 78 sessions in home and EMU settings.
              Measures seizure event detection, false alarm burden, recording quality,
              night vision coverage, and emergency response time.
            </div>
          </div>

          <div className="card shadow-sm mb-3">
            <div className="card-header fw-bold">Concept Definitions ({(defs.concepts || []).length})</div>
            <div className="card-body p-0">
              <div style={{maxHeight: 440, overflowY: 'auto'}}>
                <table className="table table-sm table-striped mb-0">
                  <thead className="table-dark sticky-top">
                    <tr>
                      <th style={{minWidth: 180}}>Concept</th>
                      <th>Description</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(defs.concepts || []).map((c, i) => (
                      <tr key={i}>
                        <td className="fw-semibold small text-primary">{c.name}</td>
                        <td className="small text-muted">{c.description}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          <div className="card shadow-sm">
            <div className="card-header fw-bold">Data Sources</div>
            <div className="card-body">
              {(defs.data_sources || []).map((src, i) => (
                <div key={i} className="small mb-1">&#x1f4c1; {src}</div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
