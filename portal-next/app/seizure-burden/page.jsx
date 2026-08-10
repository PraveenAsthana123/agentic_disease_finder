'use client';
import { useState, useEffect } from 'react';
const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const sevColor = s => s === 'Mild' ? 'success' : s === 'Moderate' ? 'warning' : s === 'Severe' ? 'danger' : 'secondary';

export default function SeizureBurdenPage() {
  const [ov, setOv]     = useState(null);
  const [bd, setBd]     = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab]   = useState('overview');
  const [expandedPt, setExpandedPt] = useState(null);

  useEffect(() => {
    fetch(`${API}/api/seizure-burden/overview`).then(r => r.json()).then(setOv).catch(() => {});
    fetch(`${API}/api/seizure-burden/breakdown`).then(r => r.json()).then(setBd).catch(() => {});
    fetch(`${API}/api/seizure-burden/definitions`).then(r => r.json()).then(setDefs).catch(() => {});
  }, []);

  if (!ov) return <div className="p-4"><div className="spinner-border text-primary" /></div>;

  const tabs = [
    { id: 'overview',   label: 'Overview' },
    { id: 'triggers',   label: 'Trigger Analysis' },
    { id: 'patients',   label: 'Patient Detail' },
    { id: 'definitions',label: 'Definitions' },
  ];

  const sevEntries = Object.entries(ov.severity_distribution || {});
  const diaryTriggers = Object.entries(ov.diary_trigger_distribution || {}).sort((a,b) => b[1]-a[1]);
  const logTriggers   = Object.entries(ov.log_trigger_distribution  || {}).sort((a,b) => b[1]-a[1]);
  const pc = ov.physiological_comparison || {};

  return (
    <div>
      <h3>Seizure Burden &amp; Trigger Dashboard</h3>
      <p className="text-muted small">
        Seizure frequency, duration, severity, and modifiable trigger analysis ·
        Sources: <code>seizure_diary</code> (25 events) + <code>seizure_trigger_logs</code> (203 days) · clinical.db
      </p>

      {/* KPI cards */}
      <div className="row mb-3">
        {[
          { label: 'Total Events',       value: ov.kpis.total_diary_events,                         color: 'primary' },
          { label: 'ER Visits',          value: ov.kpis.er_visits,                                  color: ov.kpis.er_visits > 0 ? 'danger' : 'success' },
          { label: 'Severe Events',      value: ov.kpis.severe_events,                              color: ov.kpis.severe_events > 0 ? 'danger' : 'success' },
          { label: 'Injury Events',      value: ov.kpis.injury_events,                              color: ov.kpis.injury_events > 0 ? 'warning' : 'success' },
          { label: 'Avg Duration',       value: `${ov.kpis.avg_duration_sec}s`,                     color: ov.kpis.avg_duration_sec > 120 ? 'warning' : 'info' },
          { label: 'Max Duration',       value: `${ov.kpis.max_duration_sec}s`,                     color: ov.kpis.max_duration_sec > 300 ? 'danger' : 'warning' },
          { label: 'Seizure Rate (logs)',value: `${ov.kpis.seizure_rate_in_logs_pct}%`,             color: ov.kpis.seizure_rate_in_logs_pct > 20 ? 'danger' : 'warning' },
          { label: 'Rescue Meds Used',   value: ov.kpis.rescue_med_used,                            color: ov.kpis.rescue_med_used > 0 ? 'warning' : 'success' },
        ].map(c => (
          <div key={c.label} className="col-6 col-md-3 col-lg-2 mb-2" style={{minWidth: 120}}>
            <div className="card text-center shadow-sm border-0">
              <div className="card-body py-2">
                <div className={`h4 mb-0 text-${c.color}`}>{c.value}</div>
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

      {/* ── Overview Tab ─────────────────────────────────────── */}
      {tab === 'overview' && (
        <div className="row">
          {/* Severity distribution */}
          <div className="col-md-4 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Severity Distribution</div>
              <div className="card-body">
                {sevEntries.map(([sev, cnt]) => (
                  <div key={sev} className="d-flex align-items-center mb-2">
                    <span className={`badge bg-${sevColor(sev)} me-2`} style={{minWidth: 60}}>{sev}</span>
                    <div className="flex-grow-1 me-2">
                      <div className="progress" style={{height: 20}}>
                        <div className={`progress-bar bg-${sevColor(sev)}`}
                             style={{width: `${ov.kpis.total_diary_events ? cnt / ov.kpis.total_diary_events * 100 : 0}%`}}>
                          {cnt}
                        </div>
                      </div>
                    </div>
                    <span className="small text-muted">{ov.kpis.total_diary_events ? Math.round(cnt / ov.kpis.total_diary_events * 100) : 0}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Physiological comparison */}
          <div className="col-md-8 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Physiological Profile — Seizure vs Non-Seizure Days</div>
              <div className="card-body p-0">
                <table className="table table-sm mb-0">
                  <thead>
                    <tr>
                      <th>Metric</th>
                      <th className="text-danger">Seizure Days (n={pc.seizure_days?.n})</th>
                      <th className="text-success">Non-Seizure Days (n={pc.non_seizure_days?.n})</th>
                      <th>Delta</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      { label: 'Avg Sleep (hrs)',   sz: pc.seizure_days?.avg_sleep_hours,   ns: pc.non_seizure_days?.avg_sleep_hours,   better: 'higher' },
                      { label: 'Avg Stress (1-10)', sz: pc.seizure_days?.avg_stress_level,  ns: pc.non_seizure_days?.avg_stress_level,  better: 'lower'  },
                      { label: 'Avg Fatigue (1-10)',sz: pc.seizure_days?.avg_fatigue_level, ns: pc.non_seizure_days?.avg_fatigue_level, better: 'lower'  },
                      { label: 'Missed Doses',      sz: pc.seizure_days?.avg_missed_doses,  ns: pc.non_seizure_days?.avg_missed_doses,  better: 'lower'  },
                      { label: 'Caffeine (mg)',      sz: pc.seizure_days?.avg_caffeine_mg,   ns: pc.non_seizure_days?.avg_caffeine_mg,   better: 'lower'  },
                    ].map(row => {
                      const delta = row.sz != null && row.ns != null ? (row.sz - row.ns).toFixed(1) : '—';
                      const isWorse = row.sz != null && row.ns != null &&
                        ((row.better === 'lower' && row.sz > row.ns) || (row.better === 'higher' && row.sz < row.ns));
                      return (
                        <tr key={row.label}>
                          <td className="small fw-semibold">{row.label}</td>
                          <td className={`text-danger ${isWorse ? 'fw-bold' : ''}`}>{row.sz ?? '—'}</td>
                          <td className="text-success">{row.ns ?? '—'}</td>
                          <td>
                            {delta !== '—' && (
                              <span className={`badge ${isWorse ? 'bg-danger' : 'bg-success'}`}>
                                {delta > 0 ? '+' : ''}{delta}
                              </span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>

          {/* Patient summary table */}
          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Patient Seizure Summary</div>
              <div className="card-body p-0" style={{overflowX: 'auto'}}>
                <table className="table table-sm table-striped mb-0">
                  <thead>
                    <tr>
                      <th>Patient</th><th>Events</th><th>Severe</th><th>Mild</th>
                      <th>ER Visits</th><th>Avg Duration</th><th>Max Duration</th><th>Date Range</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(ov.patient_summary || []).map((p, i) => (
                      <tr key={i}>
                        <td className="fw-semibold small">{p.patient_id}</td>
                        <td>{p.total_events}</td>
                        <td className={p.severe_count > 0 ? 'text-danger fw-bold' : ''}>{p.severe_count}</td>
                        <td>{p.mild_count}</td>
                        <td className={p.er_visits > 0 ? 'text-danger fw-bold' : ''}>{p.er_visits}</td>
                        <td>{p.avg_duration_sec}s</td>
                        <td className={p.max_duration_sec > 300 ? 'text-danger fw-bold' : ''}>{p.max_duration_sec}s</td>
                        <td className="small text-muted">
                          {p.dates.length > 0 ? `${p.dates[0]} – ${p.dates[p.dates.length - 1]}` : '—'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Trigger Analysis Tab ─────────────────────────────── */}
      {tab === 'triggers' && (
        <div className="row">
          {/* Diary triggers */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Seizure Diary — Reported Triggers</div>
              <div className="card-body">
                {diaryTriggers.map(([trigger, cnt]) => (
                  <div key={trigger} className="d-flex align-items-center mb-2">
                    <span className="small me-2" style={{minWidth: 140}}>{trigger}</span>
                    <div className="flex-grow-1 me-2">
                      <div className="progress" style={{height: 18}}>
                        <div className="progress-bar bg-warning"
                             style={{width: `${ov.kpis.total_diary_events ? cnt / ov.kpis.total_diary_events * 100 : 0}%`}}>
                          {cnt}
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Log triggers */}
          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Trigger Logs — Primary Trigger Distribution (203 days)</div>
              <div className="card-body">
                {logTriggers.filter(([t]) => t !== 'Unknown').slice(0, 10).map(([trigger, cnt]) => (
                  <div key={trigger} className="d-flex align-items-center mb-2">
                    <span className="small me-2" style={{minWidth: 160}}>{trigger.replace(/_/g, ' ')}</span>
                    <div className="flex-grow-1 me-2">
                      <div className="progress" style={{height: 18}}>
                        <div className="progress-bar bg-info"
                             style={{width: `${Math.max(5, cnt / 203 * 100)}%`}}>
                          {cnt}
                        </div>
                      </div>
                    </div>
                    <span className="small text-muted">{Math.round(cnt / 203 * 100)}%</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Duration histogram */}
          {bd && (
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-bold">Seizure Duration Histogram (trigger-log events)</div>
                <div className="card-body">
                  {(bd.seizure_duration_histogram || []).filter(b => b.count > 0).map((b, i) => (
                    <div key={i} className="d-flex align-items-center mb-2">
                      <span className="small me-2" style={{minWidth: 80}}>{b.label}</span>
                      <div className="flex-grow-1 me-2">
                        <div className="progress" style={{height: 20}}>
                          <div className={`progress-bar ${b.label.includes('>10') ? 'bg-danger' : b.label.includes('5–10') ? 'bg-warning' : 'bg-success'}`}
                               style={{width: `${Math.max(5, b.count / 53 * 100)}%`}}>
                            {b.count}
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Sleep distribution */}
          {bd && (
            <div className="col-md-6 mb-3">
              <div className="card shadow-sm">
                <div className="card-header fw-bold">Sleep Hours — Seizure vs Non-Seizure Days</div>
                <div className="card-body">
                  {Object.entries(bd.sleep_distribution?.seizure || {}).sort().map(([hr, cnt]) => (
                    <div key={hr} className="d-flex align-items-center mb-1">
                      <span className="small me-2" style={{minWidth: 30}}>{hr}</span>
                      <div className="flex-grow-1 me-1">
                        <div className="d-flex gap-1">
                          <div className="progress flex-grow-1" style={{height: 14}}>
                            <div className="progress-bar bg-danger" style={{width: `${Math.max(5, cnt / 53 * 100)}%`}} title={`Sz: ${cnt}`}>{cnt}</div>
                          </div>
                          <div className="progress flex-grow-1" style={{height: 14}}>
                            <div className="progress-bar bg-success"
                                 style={{width: `${Math.max(5, (bd.sleep_distribution?.non_seizure?.[hr] || 0) / 150 * 100)}%`}}
                                 title={`Non-Sz: ${bd.sleep_distribution?.non_seizure?.[hr] || 0}`}>
                              {bd.sleep_distribution?.non_seizure?.[hr] || 0}
                            </div>
                          </div>
                        </div>
                      </div>
                      <span className="small text-danger me-1">Sz</span>
                      <span className="small text-success">NSz</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* ── Patient Detail Tab ───────────────────────────────── */}
      {tab === 'patients' && bd && (
        <div>
          {(bd.patient_cards || []).map((pt, i) => (
            <div key={i} className="card mb-2 shadow-sm">
              <div className="card-header d-flex justify-content-between align-items-center py-2"
                   style={{cursor: 'pointer'}} onClick={() => setExpandedPt(expandedPt === i ? null : i)}>
                <span>
                  <strong>{pt.patient_id}</strong>
                  <span className={`badge bg-${pt.event_count >= 3 ? 'danger' : pt.event_count >= 2 ? 'warning' : 'info'} ms-2`}>
                    {pt.event_count} event{pt.event_count !== 1 ? 's' : ''}
                  </span>
                  <span className="text-muted small ms-2">avg {pt.avg_duration_sec}s</span>
                </span>
                <span>{expandedPt === i ? '▲' : '▼'}</span>
              </div>
              {expandedPt === i && (
                <div className="card-body p-0">
                  <table className="table table-sm mb-0">
                    <thead>
                      <tr>
                        <th>Date</th><th>Time</th><th>Duration</th><th>Severity</th>
                        <th>Aura</th><th>Awareness</th><th>Motor</th><th>Injury</th>
                        <th>Recovery</th><th>ER</th><th>Trigger</th><th>Location</th>
                      </tr>
                    </thead>
                    <tbody>
                      {pt.events.map((ev, j) => (
                        <tr key={j}>
                          <td className="small">{ev.date}</td>
                          <td className="small">{ev.time || '—'}</td>
                          <td className={ev.duration_sec > 300 ? 'text-danger fw-bold' : ''}>{ev.duration_sec ? `${ev.duration_sec}s` : '—'}</td>
                          <td><span className={`badge bg-${sevColor(ev.severity)}`}>{ev.severity}</span></td>
                          <td className="small">{ev.aura}</td>
                          <td className="small">{ev.awareness}</td>
                          <td className="small">{ev.motor_signs}</td>
                          <td className={ev.injury && ev.injury !== 'None' ? 'text-warning small fw-bold' : 'small'}>{ev.injury}</td>
                          <td className="small">{ev.recovery_min != null ? `${ev.recovery_min}m` : '—'}</td>
                          <td className={ev.er_visit === 'Yes' ? 'text-danger fw-bold small' : 'small'}>{ev.er_visit}</td>
                          <td className="small">{ev.trigger}</td>
                          <td className="small">{ev.location}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          ))}
        </div>
      )}

      {/* ── Definitions Tab ──────────────────────────────────── */}
      {tab === 'definitions' && defs && (
        <div className="row">
          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">{defs.title}</div>
              <div className="card-body">
                <p>{defs.description}</p>
                <p className="small text-muted fst-italic">{defs.clinical_context}</p>
              </div>
            </div>
          </div>

          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">KPI Definitions</div>
              <div className="card-body">
                {(defs.kpi_definitions || []).map((d, i) => (
                  <div key={i} className="mb-2">
                    <strong>{d.name}</strong>
                    <div className="small text-muted">{d.description}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Seizure Metrics</div>
              <div className="card-body">
                {(defs.seizure_metrics || []).map((d, i) => (
                  <div key={i} className="mb-2">
                    <strong>{d.name}</strong>
                    <div className="small text-muted">{d.description}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Modifiable Trigger Factors</div>
              <div className="card-body">
                {(defs.trigger_factors || []).map((d, i) => (
                  <div key={i} className="mb-2 border-bottom pb-1">
                    <strong>{d.name}</strong>
                    <div className="small text-muted">{d.description}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="col-md-6 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Severity Classification</div>
              <div className="card-body">
                {(defs.severity_levels || []).map((d, i) => (
                  <div key={i} className="mb-2">
                    <span className={`badge bg-${sevColor(d.level)} me-2`}>{d.level}</span>
                    <span className="small">{d.description}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          <div className="col-12 mb-3">
            <div className="card shadow-sm">
              <div className="card-header fw-bold">Data Sources</div>
              <div className="card-body">
                <ul className="small mb-0">
                  {(defs.data_sources || []).map((s, i) => <li key={i}>{s}</li>)}
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
