'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const SEV_COLORS = { service_crash: 'danger', server_error: 'warning', client_error: 'info', unknown: 'secondary', redirect: 'secondary', other: 'secondary' };
const LEVEL_COLORS = { ops: 'primary', system: 'info', watchdog: 'danger', git: 'success', autobuild: 'warning', ollama: 'secondary' };

export default function IncidentManagementPage() {
  const [ov, setOv]     = useState(null);
  const [bd, setBd]     = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab]   = useState('overview');
  const [levelFilter, setLevelFilter] = useState('all');

  useEffect(() => {
    fetch(`${API}/api/incident-management/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/incident-management/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/incident-management/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;
  if (!ov.available) return <div className="p-4 alert alert-warning">Incident data unavailable</div>;

  const tabs = [
    { id: 'overview',    label: 'Overview' },
    { id: 'trends',      label: 'Incidents & Trends' },
    { id: 'log',         label: 'Event Log' },
    { id: 'definitions', label: 'Definitions' },
  ];

  const maxDaily = Math.max(...(ov.daily_incident_counts || []).map(d => d.count), 1);

  return (
    <div>
      <h3>AI Incident Management</h3>
      <p className="text-muted">Real-time incident tracking from uptime.jsonl and track.jsonl</p>

      {/* KPI cards */}
      <div className="row mb-3">
        {[
          { label: 'Total Incidents',   value: ov.total_incidents,   color: 'danger' },
          { label: 'Auto-Recovery Rate', value: `${ov.auto_recovery_rate}%`, color: 'success' },
          { label: 'MTTR (min)',        value: ov.mean_time_to_recovery_minutes, color: 'info' },
          { label: 'Current Status',    value: ov.current_status,    color: ov.current_status === 'UP' ? 'success' : 'danger' },
          { label: 'Last 24h',         value: ov.incidents_last_24h, color: 'warning' },
          { label: 'Last 7d',          value: ov.incidents_last_7d,  color: 'primary' },
        ].map(c => (
          <div key={c.label} className="col-6 col-md-2 mb-2">
            <div className="card text-center shadow-sm border-0">
              <div className="card-body py-2">
                <div className={`h3 mb-0 text-${c.color}`}>{c.value}</div>
                <div className="text-muted small">{c.label}</div>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Tabs */}
      <ul className="nav nav-tabs mb-3">
        {tabs.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>{t.label}</button>
          </li>
        ))}
      </ul>

      {/* ── Overview Tab ──────────────────────────────────────── */}
      {tab === 'overview' && (
        <div className="row">
          {/* Severity Distribution */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Severity Distribution</div>
              <div className="card-body">
                {(ov.severity_distribution || []).map(s => (
                  <div key={s.level} className="d-flex justify-content-between align-items-center mb-2">
                    <span className="fw-semibold text-capitalize" style={{minWidth:130}}>{s.level.replace('_', ' ')}</span>
                    <div className="d-flex align-items-center" style={{width:'55%'}}>
                      <div className="progress flex-grow-1 me-2" style={{height:'20px'}}>
                        <div className={`progress-bar bg-${SEV_COLORS[s.level] || 'secondary'}`}
                             style={{width:`${s.percent}%`}} />
                      </div>
                      <span className="fw-bold">{s.count} ({s.percent}%)</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Daily Incident Counts */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Daily Incidents (Last 14 Days)</div>
              <div className="card-body">
                {(ov.daily_incident_counts || []).map(d => (
                  <div key={d.date} className="d-flex align-items-center mb-1">
                    <span className="text-muted small" style={{minWidth:85}}>{d.date.slice(5)}</span>
                    <div className="progress flex-grow-1 me-2" style={{height:'16px'}}>
                      <div className="progress-bar bg-danger" style={{width:`${d.count / maxDaily * 100}%`}} />
                    </div>
                    <span className="fw-bold small">{d.count}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Recent Incident Timeline */}
          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Recent Incident Timeline (Last 20)</div>
              <div className="card-body p-0">
                <table className="table table-sm table-striped mb-0">
                  <thead><tr><th>Timestamp</th><th>Event</th><th>HTTP</th><th>Recovery</th><th>TTR (min)</th></tr></thead>
                  <tbody>
                    {(ov.incident_timeline || []).map((inc, i) => (
                      <tr key={i}>
                        <td className="small">{inc.ts}</td>
                        <td><span className="badge bg-danger">{inc.event}</span></td>
                        <td><code>{inc.http}</code></td>
                        <td className="small">{inc.recovery_ts || <span className="text-muted">pending</span>}</td>
                        <td>{inc.ttr_minutes != null ? <span className="fw-bold">{inc.ttr_minutes}</span> : '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Incidents & Trends Tab ────────────────────────────── */}
      {tab === 'trends' && bd && (
        <div className="row">
          {/* Hourly Heatmap */}
          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Hourly Incident Heatmap (Last 7 Days)</div>
              <div className="card-body" style={{overflowX:'auto'}}>
                <table className="table table-sm table-bordered text-center mb-0" style={{fontSize:'0.75rem'}}>
                  <thead>
                    <tr>
                      <th></th>
                      {(bd.hourly_heatmap.hour_labels || []).map(h => <th key={h}>{h}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {(bd.hourly_heatmap.matrix || []).map((row, ri) => (
                      <tr key={ri}>
                        <td className="fw-bold">{(bd.hourly_heatmap.day_labels || [])[ri]}</td>
                        {row.map((v, ci) => {
                          const maxV = Math.max(...bd.hourly_heatmap.matrix.flat(), 1);
                          const intensity = v / maxV;
                          const bg = v === 0 ? '#f8f9fa' : `rgba(220, 53, 69, ${0.15 + intensity * 0.85})`;
                          const fg = intensity > 0.5 ? '#fff' : '#333';
                          return <td key={ci} style={{backgroundColor: bg, color: fg}}>{v || ''}</td>;
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Recovery Rate Trend */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Recovery Success Rate (Last 14 Days)</div>
              <div className="card-body">
                {(bd.recovery_success_rate_by_day || []).map(d => (
                  <div key={d.date} className="d-flex align-items-center mb-1">
                    <span className="text-muted small" style={{minWidth:85}}>{d.date.slice(5)}</span>
                    <div className="progress flex-grow-1 me-2" style={{height:'16px'}}>
                      <div className={`progress-bar bg-${d.rate >= 80 ? 'success' : d.rate >= 50 ? 'warning' : 'danger'}`}
                           style={{width:`${d.rate}%`}} />
                    </div>
                    <span className="small fw-bold">{d.rate}%</span>
                    <span className="text-muted small ms-1">({d.incidents}i/{d.recovered}r)</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* MTTR Trend */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">MTTR Trend (Last 14 Days)</div>
              <div className="card-body">
                {(() => {
                  const maxMTTR = Math.max(...(bd.mttr_trend || []).map(d => d.avg_mttr_minutes), 1);
                  return (bd.mttr_trend || []).map(d => (
                    <div key={d.date} className="d-flex align-items-center mb-1">
                      <span className="text-muted small" style={{minWidth:85}}>{d.date.slice(5)}</span>
                      <div className="progress flex-grow-1 me-2" style={{height:'16px'}}>
                        <div className="progress-bar bg-info" style={{width:`${d.avg_mttr_minutes / maxMTTR * 100}%`}} />
                      </div>
                      <span className="small fw-bold">{d.avg_mttr_minutes} min</span>
                    </div>
                  ));
                })()}
              </div>
            </div>
          </div>

          {/* Top Incident Types */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Event Types</div>
              <div className="card-body">
                {(bd.top_incident_types || []).map(t => {
                  const maxT = Math.max(...(bd.top_incident_types || []).map(x => x.count), 1);
                  return (
                    <div key={t.type} className="d-flex align-items-center mb-2">
                      <span className="fw-semibold" style={{minWidth:140}}>{t.type}</span>
                      <div className="progress flex-grow-1 me-2" style={{height:'20px'}}>
                        <div className="progress-bar bg-primary" style={{width:`${t.count / maxT * 100}%`}} />
                      </div>
                      <span className="fw-bold">{t.count}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>

          {/* Track Events by Level */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Track Events by Level</div>
              <div className="card-body">
                {(bd.track_events_by_level || []).map(l => {
                  const maxL = Math.max(...(bd.track_events_by_level || []).map(x => x.count), 1);
                  return (
                    <div key={l.level} className="d-flex align-items-center mb-2">
                      <span className={`badge bg-${LEVEL_COLORS[l.level] || 'secondary'} me-2`} style={{minWidth:80}}>{l.level}</span>
                      <div className="progress flex-grow-1 me-2" style={{height:'20px'}}>
                        <div className={`progress-bar bg-${LEVEL_COLORS[l.level] || 'secondary'}`}
                             style={{width:`${l.count / maxL * 100}%`}} />
                      </div>
                      <span className="fw-bold">{l.count}</span>
                    </div>
                  );
                })}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Event Log Tab ─────────────────────────────────────── */}
      {tab === 'log' && bd && (
        <div>
          {/* Level filter buttons */}
          <div className="mb-3">
            {['all', 'ops', 'watchdog', 'system', 'git', 'autobuild'].map(lv => (
              <button key={lv} className={`btn btn-sm me-1 ${levelFilter === lv ? 'btn-primary' : 'btn-outline-secondary'}`}
                      onClick={() => setLevelFilter(lv)}>
                {lv}
              </button>
            ))}
          </div>
          <div className="card shadow-sm">
            <div className="card-header fw-bold">Recent Track Events</div>
            <div className="card-body p-0">
              <table className="table table-sm table-striped mb-0">
                <thead><tr><th>Timestamp</th><th>Level</th><th>Event</th><th>Host</th></tr></thead>
                <tbody>
                  {(bd.recent_track_events || [])
                    .filter(e => levelFilter === 'all' || e.level === levelFilter)
                    .map((e, i) => (
                    <tr key={i}>
                      <td className="small">{e.ts}</td>
                      <td><span className={`badge bg-${LEVEL_COLORS[e.level] || 'secondary'}`}>{e.level}</span></td>
                      <td>{e.event}</td>
                      <td className="text-muted small">{e.host}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* ── Definitions Tab ───────────────────────────────────── */}
      {tab === 'definitions' && defs && (
        <div className="card shadow-sm">
          <div className="card-header fw-bold">Metric Definitions</div>
          <div className="card-body p-0">
            <table className="table table-sm mb-0">
              <thead><tr><th style={{width:'25%'}}>Term</th><th>Definition</th></tr></thead>
              <tbody>
                {(defs.metrics || []).map((m, i) => (
                  <tr key={i}>
                    <td className="fw-semibold">{m.term}</td>
                    <td>{m.definition}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
