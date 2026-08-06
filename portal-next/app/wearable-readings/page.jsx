'use client';
import { useState, useEffect } from 'react';

const API = process.env.NEXT_PUBLIC_API || 'http://localhost:8010';

const riskColor = r => r >= 40 ? 'danger' : r >= 30 ? 'warning' : 'success';
const healthColor = h => h >= 70 ? 'success' : h >= 50 ? 'warning' : 'danger';
const hrColor = hr => hr >= 90 ? 'danger' : hr >= 80 ? 'warning' : 'success';

export default function WearableReadingsDashboard() {
  const [ov, setOv] = useState(null);
  const [bd, setBd] = useState(null);
  const [defs, setDefs] = useState(null);
  const [tab, setTab] = useState('overview');
  const [search, setSearch] = useState('');
  const [sortBy, setSortBy] = useState('seizure_risk');
  const [err, setErr] = useState(null);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/wearable-readings/overview`).then(r => r.json()),
      fetch(`${API}/api/wearable-readings/breakdown`).then(r => r.json()),
      fetch(`${API}/api/wearable-readings/definitions`).then(r => r.json()),
    ]).then(([o, b, d]) => { setOv(o); setBd(b); setDefs(d); })
      .catch(e => setErr(String(e)));
  }, []);

  if (err) return <div className="alert alert-danger m-3">Failed to load: {err}</div>;
  if (!ov) return <div className="text-muted p-3">Loading wearable readings data…</div>;

  const TABS = [
    { id: 'overview',   label: '📊 Overview' },
    { id: 'patients',   label: '👤 Per Patient' },
    { id: 'highrisk',  label: '🚨 High Risk' },
    { id: 'recent',    label: '🕐 Recent Readings' },
    { id: 'definitions', label: '📖 Definitions' },
  ];

  const patients = bd?.per_patient || [];
  const filtered = patients.filter(p =>
    !search || p.patient_id.toLowerCase().includes(search.toLowerCase()) ||
    p.device_id.toLowerCase().includes(search.toLowerCase())
  ).sort((a, b) => {
    if (sortBy === 'seizure_risk')  return b.avg_seizure_risk - a.avg_seizure_risk;
    if (sortBy === 'seizure_events') return b.seizure_events - a.seizure_events;
    if (sortBy === 'health')        return a.avg_health - b.avg_health;
    if (sortBy === 'heart_rate')    return b.avg_hr - a.avg_hr;
    return a.patient_id.localeCompare(b.patient_id);
  });

  const highRisk = bd?.high_risk_patients || [];
  const recentReadings = bd?.recent_readings || [];
  const seizureEvents = bd?.seizure_events || [];
  const glossary = defs?.glossary || [];

  return (
    <div className="p-3">
      <h3>⌚ Wearable Readings Dashboard</h3>
      <p className="text-muted">
        Continuous biometric monitoring — {ov.total_readings} readings ·{' '}
        {ov.total_patients} patients · {ov.total_devices} devices ·{' '}
        {ov.seizure_event_count} seizure events · {ov.fall_event_count} fall events ·{' '}
        detection rate {ov.seizure_detection_rate}%
      </p>

      <ul className="nav nav-tabs mb-3">
        {TABS.map(t => (
          <li key={t.id} className="nav-item">
            <button className={`nav-link ${tab === t.id ? 'active' : ''}`} onClick={() => setTab(t.id)}>
              {t.label}
            </button>
          </li>
        ))}
      </ul>

      {/* ── OVERVIEW ── */}
      {tab === 'overview' && (
        <div>
          {/* KPI cards */}
          <div className="row mb-3">
            {[
              ['Total Readings', ov.total_readings, 'primary'],
              ['Patients', ov.total_patients, 'info'],
              ['Devices', ov.total_devices, 'secondary'],
              ['Seizure Events', ov.seizure_event_count, 'danger'],
              ['Fall Events', ov.fall_event_count, 'warning'],
              ['Detection Rate', `${ov.seizure_detection_rate}%`, 'dark'],
            ].map(([label, val, c]) => (
              <div key={label} className="col-6 col-md-2 mb-2">
                <div className={`card border-${c} text-center`}>
                  <div className="card-body py-2">
                    <div className="fw-bold fs-5">{val}</div>
                    <div className="text-muted small">{label}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          {/* Vitals averages */}
          <div className="row mb-3">
            {[
              ['Avg Heart Rate', `${ov.avg_heart_rate} bpm`, 'danger'],
              ['Avg Steps/Day', ov.avg_steps?.toLocaleString(), 'success'],
              ['Avg Sleep', `${ov.avg_sleep_hours} h`, 'primary'],
              ['Avg SpO2', `${ov.avg_spo2}%`, 'info'],
              ['Avg Health Score', ov.avg_health_score, 'success'],
              ['Avg Stress Score', ov.avg_stress_score, 'warning'],
              ['Avg Seizure Risk', ov.avg_seizure_risk, 'danger'],
            ].map(([label, val, c]) => (
              <div key={label} className="col-6 col-md-3 mb-2">
                <div className={`card border-${c}`}>
                  <div className="card-body py-2">
                    <div className="small text-muted">{label}</div>
                    <div className={`fw-bold text-${c}`}>{val}</div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <div className="row">
            {/* Heart Rate Distribution */}
            <div className="col-md-4 mb-3">
              <div className="card h-100">
                <div className="card-header fw-bold">💓 Heart Rate Distribution</div>
                <div className="card-body">
                  {(ov.heart_rate_distribution || []).map(b => {
                    const pct = Math.round((b.count / ov.total_readings) * 100);
                    const col = b.bucket === '90+' ? 'danger' : b.bucket === '80-90' ? 'warning' : 'success';
                    return (
                      <div key={b.bucket} className="mb-2">
                        <div className="d-flex justify-content-between small mb-1">
                          <span>{b.bucket} bpm</span>
                          <span className="text-muted">{b.count} ({pct}%)</span>
                        </div>
                        <div className="progress" style={{ height: '10px' }}>
                          <div className={`progress-bar bg-${col}`} style={{ width: `${pct}%` }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Activity Distribution */}
            <div className="col-md-4 mb-3">
              <div className="card h-100">
                <div className="card-header fw-bold">🏃 Activity Distribution</div>
                <div className="card-body">
                  {(ov.activity_distribution || []).map(a => {
                    const pct = Math.round((a.count / ov.total_readings) * 100);
                    const col = a.category === 'active' ? 'success' : a.category === 'moderate' ? 'info' : a.category === 'light' ? 'warning' : 'secondary';
                    return (
                      <div key={a.category} className="mb-2">
                        <div className="d-flex justify-content-between small mb-1">
                          <span className="text-capitalize">{a.category}</span>
                          <span className="text-muted">{a.count} ({pct}%)</span>
                        </div>
                        <div className="progress" style={{ height: '10px' }}>
                          <div className={`progress-bar bg-${col}`} style={{ width: `${pct}%` }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Sleep Quality Distribution */}
            <div className="col-md-4 mb-3">
              <div className="card h-100">
                <div className="card-header fw-bold">😴 Sleep Quality Distribution</div>
                <div className="card-body">
                  <div className="d-flex flex-wrap gap-1">
                    {(ov.sleep_quality_distribution || []).map(s => {
                      const col = s.score >= 8 ? 'success' : s.score >= 5 ? 'warning' : 'danger';
                      return (
                        <div key={s.score} className="text-center" style={{ minWidth: '50px' }}>
                          <span className={`badge bg-${col}`}>{s.score}/10</span>
                          <div className="small text-muted">{s.count}</div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Daily trend */}
          {ov.daily_trend && ov.daily_trend.length > 0 && (
            <div className="card mb-3">
              <div className="card-header fw-bold">📅 Daily Trend (last {ov.daily_trend.length} days)</div>
              <div className="card-body p-0">
                <div style={{ overflowX: 'auto' }}>
                  <table className="table table-sm table-striped mb-0">
                    <thead>
                      <tr>
                        <th>Date</th>
                        <th>Avg HR (bpm)</th>
                        <th>Avg Steps</th>
                        <th>Avg Sleep (h)</th>
                        <th>Seizure Events</th>
                      </tr>
                    </thead>
                    <tbody>
                      {ov.daily_trend.map(d => (
                        <tr key={d.date}>
                          <td>{d.date}</td>
                          <td><span className={`badge bg-${hrColor(d.avg_heart_rate)}`}>{d.avg_heart_rate}</span></td>
                          <td>{d.avg_steps?.toLocaleString()}</td>
                          <td>{d.avg_sleep}</td>
                          <td>
                            {d.seizure_events > 0
                              ? <span className="badge bg-danger">{d.seizure_events}</span>
                              : <span className="text-muted">—</span>}
                          </td>
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

      {/* ── PER PATIENT ── */}
      {tab === 'patients' && (
        <div>
          <div className="row mb-2 g-2 align-items-end">
            <div className="col-md-5">
              <input
                className="form-control form-control-sm"
                placeholder="Search patient or device…"
                value={search}
                onChange={e => setSearch(e.target.value)}
              />
            </div>
            <div className="col-md-4">
              <select className="form-select form-select-sm" value={sortBy} onChange={e => setSortBy(e.target.value)}>
                <option value="seizure_risk">Sort: Seizure Risk ↓</option>
                <option value="seizure_events">Sort: Seizure Events ↓</option>
                <option value="health">Sort: Health Score ↑</option>
                <option value="heart_rate">Sort: Heart Rate ↓</option>
                <option value="patient_id">Sort: Patient ID</option>
              </select>
            </div>
            <div className="col-md-3 text-muted small">{filtered.length} / {patients.length} patients</div>
          </div>
          <div style={{ overflowX: 'auto' }}>
            <table className="table table-sm table-hover">
              <thead className="table-light">
                <tr>
                  <th>Patient</th>
                  <th>Device</th>
                  <th>Readings</th>
                  <th>Avg HR</th>
                  <th>Steps</th>
                  <th>Sleep (h)</th>
                  <th>SpO2 (%)</th>
                  <th>Health Score</th>
                  <th>Seizure Events</th>
                  <th>Fall Events</th>
                  <th>Seizure Risk</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map(p => (
                  <tr key={p.patient_id}>
                    <td><strong>{p.patient_id}</strong></td>
                    <td className="text-muted small">{p.device_id}</td>
                    <td>{p.readings_count}</td>
                    <td><span className={`badge bg-${hrColor(p.avg_hr)}`}>{p.avg_hr}</span></td>
                    <td>{p.avg_steps?.toLocaleString()}</td>
                    <td>{p.avg_sleep}</td>
                    <td>{p.avg_spo2}</td>
                    <td><span className={`badge bg-${healthColor(p.avg_health)}`}>{p.avg_health}</span></td>
                    <td>
                      {p.seizure_events > 0
                        ? <span className="badge bg-danger">{p.seizure_events}</span>
                        : <span className="text-muted">0</span>}
                    </td>
                    <td>
                      {p.fall_events > 0
                        ? <span className="badge bg-warning text-dark">{p.fall_events}</span>
                        : <span className="text-muted">0</span>}
                    </td>
                    <td><span className={`badge bg-${riskColor(p.avg_seizure_risk)}`}>{p.avg_seizure_risk}%</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── HIGH RISK ── */}
      {tab === 'highrisk' && (
        <div>
          <h6 className="mb-3">🚨 High-Risk Patients (elevated seizure risk or events)</h6>
          <div className="row mb-4">
            {highRisk.map(p => (
              <div key={p.patient_id} className="col-md-4 mb-3">
                <div className="card border-danger">
                  <div className="card-header bg-danger text-white d-flex justify-content-between">
                    <span>{p.patient_id}</span>
                    <span className="small">{p.device_id}</span>
                  </div>
                  <div className="card-body">
                    <div className="row text-center">
                      {[
                        ['Seizure Risk', `${p.avg_seizure_risk}%`, 'danger'],
                        ['Seizure Events', p.seizure_events, 'danger'],
                        ['Fall Events', p.fall_events, 'warning'],
                        ['Health Score', p.avg_health, p.avg_health >= 70 ? 'success' : 'warning'],
                        ['Avg HR', `${p.avg_hr}`, 'primary'],
                        ['SpO2', `${p.avg_spo2}%`, 'info'],
                      ].map(([label, val, c]) => (
                        <div key={label} className="col-6 mb-2">
                          <div className="small text-muted">{label}</div>
                          <div className={`fw-bold text-${c}`}>{val}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>

          <h6 className="mb-3">⚡ Recent Seizure Events (wearable-detected)</h6>
          <div style={{ overflowX: 'auto' }}>
            <table className="table table-sm table-hover">
              <thead className="table-light">
                <tr>
                  <th>Patient</th>
                  <th>Date</th>
                  <th>Confidence</th>
                  <th>Heart Rate (avg)</th>
                </tr>
              </thead>
              <tbody>
                {seizureEvents.map((e, i) => (
                  <tr key={i}>
                    <td><strong>{e.patient_id}</strong></td>
                    <td>{e.date}</td>
                    <td>
                      <span className={`badge bg-${e.confidence >= 0.75 ? 'danger' : 'warning'}`}>
                        {(e.confidence * 100).toFixed(0)}%
                      </span>
                    </td>
                    <td><span className={`badge bg-${hrColor(e.heart_rate_avg)}`}>{e.heart_rate_avg} bpm</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── RECENT READINGS ── */}
      {tab === 'recent' && (
        <div>
          <h6 className="mb-3">🕐 Most Recent Readings ({recentReadings.length} records)</h6>
          {recentReadings.map((r, i) => (
            <div key={i} className="card mb-2">
              <div className="card-header d-flex justify-content-between align-items-center py-2">
                <span className="fw-bold">{r.patient_id} — {r.device_id}</span>
                <span className="text-muted small">{r.reading_date}</span>
                <div>
                  {r.seizure_detected && <span className="badge bg-danger ms-1">Seizure Detected</span>}
                  {r.fall_detected && <span className="badge bg-warning text-dark ms-1">Fall Detected</span>}
                </div>
              </div>
              <div className="card-body py-2">
                <div className="row">
                  {/* Vitals */}
                  <div className="col-md-4">
                    <div className="small text-muted fw-bold mb-1">VITALS</div>
                    <div className="row">
                      {[
                        ['HR avg', `${r.heart_rate_avg} bpm`],
                        ['HR range', `${r.heart_rate_min}–${r.heart_rate_max}`],
                        ['HRV', `${r.heart_rate_variability} ms`],
                        ['Resting HR', `${r.resting_heart_rate} bpm`],
                        ['SpO2', `${r.spo2}%`],
                        ['Skin Temp', `${r.skin_temperature}°C`],
                      ].map(([l, v]) => (
                        <div key={l} className="col-6 mb-1">
                          <span className="text-muted small">{l}: </span>
                          <span className="small fw-bold">{v}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                  {/* Activity */}
                  <div className="col-md-4">
                    <div className="small text-muted fw-bold mb-1">ACTIVITY</div>
                    <div className="row">
                      {[
                        ['Steps', r.steps?.toLocaleString()],
                        ['Distance', `${r.distance_km} km`],
                        ['Calories', r.calories_burned],
                        ['Active min', r.active_minutes],
                      ].map(([l, v]) => (
                        <div key={l} className="col-6 mb-1">
                          <span className="text-muted small">{l}: </span>
                          <span className="small fw-bold">{v}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                  {/* Sleep & Risk */}
                  <div className="col-md-4">
                    <div className="small text-muted fw-bold mb-1">SLEEP & RISK</div>
                    <div className="row">
                      {[
                        ['Sleep', `${r.sleep_duration_hours} h`],
                        ['Quality', `${r.sleep_quality_score}/100`],
                        ['Deep', `${r.deep_sleep_pct}%`],
                        ['REM', `${r.rem_sleep_pct}%`],
                        ['Stress', r.stress_score],
                        ['Health Score', r.health_score],
                        ['Seizure Risk', `${r.seizure_risk_score}%`],
                      ].map(([l, v]) => (
                        <div key={l} className="col-6 mb-1">
                          <span className="text-muted small">{l}: </span>
                          <span className="small fw-bold">{v}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
                {r.seizure_detected && (
                  <div className="alert alert-danger py-1 mt-2 small mb-0">
                    ⚡ Seizure detected — confidence {(r.seizure_detection_confidence * 100).toFixed(0)}%
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ── DEFINITIONS ── */}
      {tab === 'definitions' && (
        <div>
          <h6 className="mb-3">📖 Clinical Glossary — Wearable Biometrics</h6>
          <div className="row">
            {glossary.map((g, i) => (
              <div key={i} className="col-md-6 mb-3">
                <div className="card h-100">
                  <div className="card-header fw-bold py-2">{g.term}</div>
                  <div className="card-body py-2">
                    <p className="small mb-0">{g.definition}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
